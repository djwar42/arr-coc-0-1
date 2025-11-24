# Advanced Security

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/enterprise">Team & Enterprise</a> plans.

Enterprise Hub organizations can improve their security with advanced security controls for both members and repositories.

<div class="flex justify-center" style="max-width: 550px">
    <img class="block dark:hidden m-0!" src="https://cdn-uploads.huggingface.co/production/uploads/5dd96eb166059660ed1ee413/LqAmGSG7YbP7Y8vJJr7NQ.png" alt="screenshot of the Organization settings page for Advanced security."/>
    <img class="hidden dark:block m-0!" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/advanced-security-dark.png" alt="screenshot of the Organization settings page for Advanced security in dark mode."/>
</div>

## Members Security

Configure additional security settings to protect your organization:

- **Two-Factor Authentication (2FA)**: Require all organization members to enable 2FA for enhanced account security.
- **User Approval**: For organizations with a verified domain name, require admin approval for new users with matching email addresses. This adds a verified badge to your organization page.

## Repository Visibility Controls

Manage the default visibility of repositories in your organization:

- **Public by default**: New repositories are created with public visibility
- **Private by default**: New repositories are created with private visibility. Note that changing this setting will not affect existing repositories.
- **Private only**: Enforce private visibility for all new repositories, with only organization admins able to change visibility settings

These settings help organizations maintain control of their ownership while enabling collaboration when needed.

# Advanced Single Sign-On (SSO)

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/contact/sales?from=enterprise" target="_blank">Enterprise Plus</a> plan.

Advanced Single Sign-On (SSO) capabilities extend the standard [SSO features](./security-sso) available in the Enterprise Hub, offering enhanced control and automation for user management and access across the entire Hugging Face platform for your organization members.

## User Provisioning

Advanced SSO introduces automated user provisioning, which simplifies the onboarding and offboarding of users.

*   **Just-In-Time (JIT) Provisioning**: When a user from your organization attempts to log in to Hugging Face for the first time via SSO, an account can be automatically created for them if one doesn't already exist. Their profile information and role mappings can be populated based on attributes from your IdP.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/sso/jit-flow-chart.png"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/sso/jit-flow-chart-dark.png"/>
</div>

*   **System for Cross-domain Identity Management (SCIM)**: For more robust user lifecycle management, SCIM allows your IdP to communicate user identity information to Hugging Face. This enables automatic creation, updates (e.g., name changes, role changes), and deactivation of user accounts on Hugging Face as changes occur in your IdP. This ensures that user access is always up-to-date with their status in your organization.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/sso/scim-flow-chart.png"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/sso/scim-flow-chart-dark.png"/>
</div>

Learn more about how to set up and manage SCIM in our [dedicated guide](./enterprise-hub-scim).

## Global SSO Enforcement

Beyond gating access to specific organizational content, Advanced SSO can be configured to make your IdP the mandatory authentication route for all your organization's members interacting with any part of the Hugging Face platform. Your organization's members will be required to authenticate via your IdP for all Hugging Face services, not just when accessing private or organizational repositories.

This feature is particularly beneficial for organizations requiring a higher degree of control, security, and automation in managing their users on Hugging Face.

## Limitations on Managed User Accounts

> [!WARNING]
> Important Considerations for Managed Accounts.

To ensure organizational control and data governance, user accounts provisioned and managed via Advanced SSO ("managed user accounts") have specific limitations:

*   **No Public Content Creation**: Managed user accounts cannot create public content on the Hugging Face platform. This includes, but is not limited to, public models, datasets, or Spaces. All content created by these accounts is restricted to within your organization or private visibility.
*   **No External Collaboration**: Managed user accounts are restricted from collaborating outside of your Hugging Face organization. This means they cannot, for example, join other organizations, contribute to repositories outside their own organization.

These restrictions are in place to maintain the integrity and security boundaries defined by your enterprise. If members of your organization require the ability to create public content or collaborate more broadly on the Hugging Face platform, they will need to do so using a separate, personal Hugging Face account that is not managed by your organization's Advanced SSO.

# Analytics

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/enterprise">Team & Enterprise</a> plans.

## Publisher Analytics Dashboard

Track all your repository activity with a detailed downloads overview that shows total downloads for all the Models and Datasets published by your organization. Toggle between "All Time" and "Last Month" views to gain insights across your repositories over different periods.

<div class="flex justify-center" style="max-width: 550px">
<img class="block dark:hidden m-0!" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise-analytics.png" alt="screenshot of the Dataset Viewer on a private dataset owned by an Enterprise Hub organization."/>
<img class="hidden dark:block m-0!" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise-analytics-dark.png" alt="screenshot of the Dataset Viewer on a private dataset owned by an Enterprise Hub organization."/>
</div>

### Per-repo breakdown

Explore the metrics of individual repositories with the per-repository drill-down table. Utilize the built-in search feature to quickly locate specific repositories. Each row also features a time-series graph that illustrates the trend of downloads over time.

## Export Publisher Analytics as CSV

Download a comprehensive CSV file containing analytics for all your repositories, including model and dataset download activity.

### Response Structure

The CSV file is made of daily download records for each of your models and datasets.

```csv
repoType,repoName,total,timestamp,downloads
model,huggingface/CodeBERTa-small-v1,4362460,2021-01-22T00:00:00.000Z,4
model,huggingface/CodeBERTa-small-v1,4362460,2021-01-23T00:00:00.000Z,7
model,huggingface/CodeBERTa-small-v1,4362460,2021-01-24T00:00:00.000Z,2
dataset,huggingface/documentation-images,2167284,2021-11-27T00:00:00.000Z,3
dataset,huggingface/documentation-images,2167284,2021-11-28T00:00:00.000Z,18
dataset,huggingface/documentation-images,2167284,2021-11-29T00:00:00.000Z,7
```

### Repository Object Structure

Each record in the CSV contains:

- `repoType`: The type of repository (e.g., "model", "dataset")
- `repoName`: Full repository name including organization (e.g., "huggingface/documentation-images")
- `total`: Cumulative number of downloads for this repository
- `timestamp`: ISO 8601 formatted date (UTC)
- `downloads`: Number of downloads for that day

Records are ordered chronologically and provide a daily granular view of download activity for each repository.

# Datasets

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/enterprise">Team & Enterprise</a> plans.

Data Studio is enabled on private datasets under your Enterprise Hub organization.

Data Studio allows teams to understand their data and to help them build better data processing and filtering for AI. This powerful viewer allows you to explore dataset content, inspect data distributions, filter by values, search for keywords, or even run SQL queries on your data without leaving your browser.

More information about [Data Studio](./datasets-viewer).

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/private-dataset-viewer.png" alt="screenshot of Data Studio on a private dataset owned by an Enterprise Hub organization."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/private-dataset-viewer-dark.png" alt="screenshot of Data Studio on a private dataset owned by an Enterprise Hub organization."/>
</div>

# Gating Group Collections

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/enterprise">Team & Enterprise</a> plans.

Gating Group Collections allow organizations to grant (or reject) access to all the models and datasets in a collection at once, rather than per repo. Users will only have to go through **a single access request**.

To enable Gating Group in a collection:

- the collection owner must be an organization
- the organization must be subscribed to a Team or Enterprise plan
- all models and datasets in the collection must be owned by the same organization as the collection
- each model or dataset in the collection may only belong to one Gating Group Collection (but they can still be included in non-gating i.e. _regular_ collections).

> [!TIP]
> Gating only applies to models and datasets; any other resource part of the collection (such as a Space or a Paper) won't be affected.

## Manage gating group as an organization admin

To enable access requests, go to the collection page and click on **Gating group** in the bottom-right corner.

<div class="flex justify-center" style="max-width: 750px">
    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-collection-disabled.webp"
        alt="Hugging Face collection page with gating group collection feature disabled"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-collection-disabled.webp"
        alt="Hugging Face collection page with gating group collection feature disabled"
    />
</div>

By default, gating group is disabled: click on **Configure Access Requests** to open the settings

<div class="flex justify-center" style="max-width: 750px">
    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-modal-disabled.webp"
        alt="Hugging Face gating group collection settings with gating disabled"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-modal-disabled.webp"
        alt="Hugging Face gating group collection settings with gating disabled"
    />
</div>

By default, access to the repos in the collection is automatically granted to users when they request it. This is referred to as **automatic approval**. In this mode, any user can access your repos once they’ve agreed to share their contact information with you.

<div class="flex justify-center" style="max-width: 750px">
    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-modal-enabling.webp"
        alt="Hugging Face gating group collection settings with automatic mode selected"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-modal-enabling.webp"
        alt="Hugging Face gating group collection settings with automatic mode selected"
    />
</div>

If you want to manually approve which users can access repos in your collection, you must set it to **Manual Review**. When this is the case, you will notice a new option:

**Notifications frequency**, which lets you configure when to get notified about new users requesting access. It can be set to once a day or real-time. By default, emails are sent to the first 5 admins of the organization. You can also set a different email address in the **Notifications email** field.

<div class="flex justify-center" style="max-width: 750px">
    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-modal-manual.webp"
        alt="Hugging Face gating group collection settings with manual review mode selected"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-modal-manual.webp"
        alt="Hugging Face gating group collection settings with manual review mode selected"
    />
</div>

### Review access requests

Once access requests are enabled, you have full control of who can access repos in your gating group collection, whether the approval mode is manual or automatic. You can review and manage requests either from the UI or via the API.

**Approving a request for a repo in a gating group collection will automatically approve access to all repos (models and datasets) in that collection.**

#### From the UI

You can review who has access to all the repos in your Gating Group Collection from the settings page of any of the repos in the collection, by clicking on the **Review access requests** button:

<div class="flex justify-center" style="max-width: 750px">
    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-repo-settings.webp"
        alt="Hugging Face repo access settings when repo is in a gating group collection"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-repo-settings.webp"
        alt="Hugging Face repo access settings when repo is in a gating group collection"
    />
</div>

This will open a modal with 3 lists of users:

- **pending**: the list of users waiting for approval to access your repository. This list is empty unless you’ve selected **Manual Review**. You can either **Accept** or **Reject** each request. If the request is rejected, the user cannot access your repository and cannot request access again.
- **accepted**: the complete list of users with access to your repository. You can choose to **Reject** access at any time for any user, whether the approval mode is manual or automatic. You can also **Cancel** the approval, which will move the user to the **pending** list.
- **rejected**: the list of users you’ve manually rejected. Those users cannot access your repositories. If they go to your repository, they will see a message _Your request to access this repo has been rejected by the repo’s authors_.

<div class="flex justify-center" style="max-width: 750px">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/models-gated-enabled-pending-users.png"
        alt="Manage access requests modal for a repo in a gating group collection"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/models-gated-enabled-pending-users-dark.png"
        alt="Manage access requests modal for a repo in a gating group collection"
    />

</div>

#### Via the API

You can programmatically manage access requests in a Gated Group Collection through the API of any of its models or datasets.

Visit our [gated models](https://huggingface.co/docs/hub/models-gated#via-the-api) or [gated datasets](https://huggingface.co/docs/hub/datasets-gated#via-the-api) documentation to know more about it.

#### Download access report

You can download access reports for the Gated Group Collection through the settings page of any of its models or datasets.

Visit our [gated models](https://huggingface.co/docs/hub/models-gated#download-access-report) or [gated datasets](https://huggingface.co/docs/hub/datasets-gated#download-access-report) documentation to know more about it.

#### Customize requested information

Organizations can customize the gating parameters as well as the user information that is collected per gated repo. Please, visit our [gated models](https://huggingface.co/docs/hub/models-gated#customize-requested-information) or [gated datasets](https://huggingface.co/docs/hub/datasets-gated#customize-requested-information) documentation for more details.

> [!WARNING]
> There is currently no way to customize the gate parameters and requested information in a centralized way. If you want to collect the same data no matter which collection's repository a user requests access throughout, you need to add the same gate parameters in the metadata of all the models and datasets of the collection, and keep it synced.

## Access gated repos in a Gating Group Collection as a user

A Gated Group Collection shows a specific icon next to its name:

<div class="flex justify-center" style="max-width: 750px">
    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-collection-enabled.webp"
        alt="Hugging Face collection page with gating group collection feature enabled"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-collection-enabled.webp"
        alt="Hugging Face collection page with gating group collection feature enabled"
    />
</div>

To get access to the models and datasets in a Gated Group Collection, a single access request on the page of any of those repositories is needed. Once your request is approved, you will be able to access all the other repositories in the collection, including future ones.

Visit our [gated models](https://huggingface.co/docs/hub/models-gated#access-gated-models-as-a-user) or [gated datasets](https://huggingface.co/docs/hub/datasets-gated#access-gated-datasets-as-a-user) documentation to learn more about requesting access to a repository.

# Network Security

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/contact/sales?from=enterprise" target="_blank">Enterprise Plus</a> plan.

## Define your organization IP Ranges

You can list the IP addresses of your organization's outbound traffic to apply for higher rate limits and/or to enforce authenticated access to Hugging Face from your corporate network.
The outbound IP address ranges are defined in <a href="https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing" target="_blank">CIDR</a> format. For example, `52.219.168.0/24` or `2600:1f69:7400::/40`.

You can set multiple ranges, one per line.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/network-sec-ip-ranges.png" alt="Screenshot of the Organization IP Ranges field."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-network-sec-ip-ranges.png" alt="Screenshot of the Organization IP Ranges field."/>
</div>


## Higher Hub Rate Limits

Most of the actions on the Hub have limits; for example, users are limited to creating a certain number of repositories per day. Enterprise Plus automatically gives your users the highest rate limits possible for every action.

Additionally, once your IP ranges are set, enabling the "Higher Hub Rate Limits" option allows your organization to benefit from the highest HTTP rate limits on the Hub API, unlocking large volumes of model or dataset downloads.

For more information about rate limits, see the [Hub Rate limits](./rate-limits) documentation.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/network-sec-rate-limit.png" alt="Screenshot of the toggle to enable High rate-limits."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-network-sec-rate-limit.png" alt="Screenshot of the toggle to enable High rate-limits."/>
</div>


## Restrict organization access to your IP ranges only

This option restricts access to your organization's resources to only those coming from your defined IP ranges. No one can access your organization resources outside your IP ranges. The rules also apply to access tokens. When enabled, this option unlocks additional nested security settings below.


### Require login for users in your IP ranges

When this option is enabled, anyone visiting Hugging Face from your corporate network must be logged in and belong to your organization (requires a manual verification when IP ranges have changed). If enabled, you can optionally define a content access policy.

All public pages will show the following message if access is unauthenticated:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/network-sec-restricted-url.png" alt="Screenshot of restricted pages on the Hub."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-network-sec-restricted-url.png" alt="Screenshot of restricted pages on the Hub."/>
</div>



### Content Access Policy

Define a fine-grained Content Access Policy by blocking certain sections of the Hugging Face Hub.

For example, you can block your organization's members from accessing Spaces by adding `/spaces/*` to the blocked URLs. When users of your organization navigate to a page that matches the URL pattern, they'll be presented the following page:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/network-sec-blocked-url.png" alt="Screenshot of blocked pages on the Hub."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-network-sec-blocked-url.png" alt="Screenshot of blocked pages on the Hub."/>
</div>

To define Blocked URLs, enter URL patterns, without the domain name, one per line:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/network-sec-cap.png" alt="Screenshot of blocked pages on the Hub."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-network-sec-cap.png" alt="Screenshot of blocked pages on the Hub."/>
</div>

The Allowed URLs field, enables you to define some exception to the blocking rules, especially. For example by allowing a specific URL within the Blocked URLs pattern, ie `/spaces/meta-llama/*`

# Resource groups

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/enterprise">Team & Enterprise</a> plans.

Resource Groups allow organizations to enforce fine-grained access control to their repositories.

<div class="flex justify-center" style="max-width: 550px">
  <img
    class="block dark:hidden m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/resource-groups.png"
    alt="screenshot of Hugging Face Single Sign-On (SSO) feature"
  />
  <img
    class="hidden dark:block m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-resource-groups.png"
    alt="screenshot of Hugging Face Single Sign-On (SSO) feature"
  />
</div>

This feature allows organization administrators to:

- Group related repositories together for better organization
- Control member access at a group level rather than individual repository level
- Assign different permission roles (read, contributor, write, admin) to team members
- Keep private repositories visible only to authorized group members
- Enable multiple teams to work independently within the same organization

This Enterprise Hub feature helps organizations manage complex team structures and maintain proper access control over their repositories.

[Getting started with Resource Groups →](./security-resource-groups)

# User Provisioning (SCIM)

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/contact/sales?from=enterprise" target="_blank">Enterprise Plus</a> plan.

SCIM, or System for Cross-domain Identity Management, is a standard for automating user provisioning. It allows you to connect your Identity Provider (IdP) to Hugging Face to automatically manage your organization's members.

With SCIM, you can:
- **Provision users**: Automatically create user accounts in your Hugging Face organization when they are assigned the application in your IdP.
- **Update user attributes**: Changes made to user profiles in your IdP (like name or email) are automatically synced to Hugging Face.
- **Provision groups**: Create groups in your Hugging Face organization based on groups in your IdP.
- **Deprovision users**: Automatically deactivate user accounts in your Hugging Face organization when they are unassigned from the application or deactivated in your IdP.

This ensures that your Hugging Face organization's member list is always in sync with your IdP, streamlining user lifecycle management and improving security.

## How to enable SCIM

To enable SCIM, go to your organization's settings, navigate to the **SSO** tab, and then select the **SCIM** sub-tab.

You will find the **SCIM Tenant URL** and a button to generate an **access token**. You will need both of these to configure your IdP. The SCIM token is a secret and should be stored securely in your IdP's configuration.

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/sso/scim-settings.png"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/sso/scim-settings-dark.png"/>
</div>

Once SCIM is enabled in your IdP, users and groups provisioned will appear in the "Users Management" and "SCIM" tabs respectively.

## Supported Identity Providers

We support SCIM with any IdP that implements the SCIM 2.0 protocol. We have specific guides for some of the most popular providers:
- [How to configure SCIM with Microsoft Entra ID](./security-sso-entra-id-scim)
- [How to configure SCIM with Okta](./security-sso-okta-scim)

# Tokens Management

> [!WARNING]
> This feature is part of the <a href="https://huggingface.co/enterprise">Team & Enterprise</a> plans.

Tokens Management enables organization administrators to oversee access tokens within their organization, ensuring secure access to organization resources.

## Viewing and Managing Access Tokens

The token listing feature displays all access tokens within your organization. Administrators can:

- Monitor token usage and identify or prevent potential security risks:
  - Unauthorized access to private resources ("leaks")
  - Overly broad access scopes
  - Suboptimal token hygiene (e.g., tokens that have not been rotated in a long time)
- Identify and revoke inactive or unused tokens

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tokens-management-list.png" />
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tokens-management-list-dark.png" />
</div>

Fine-grained tokens display their specific permissions:

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tokens-management-detail.png" />
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tokens-management-detail-dark.png" />
</div>

## Token Policy

Enterprise organization administrators can enforce the following policies:

| **Policy**                                        | **Unscoped (Read/Write) Access Tokens** | **Fine-Grained Tokens**                                     |
| ------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------- |
| **Allow access via User Access Tokens (default)** | Authorized                              | Authorized                                                  |
| **Only access via fine-grained tokens**           | Unauthorized                            | Authorized                                                  |
| **Do not require administrator approval**         | Unauthorized                            | Authorized                                                  |
| **Require administrator approval**                | Unauthorized                            | Unauthorized without an approval (except for admin-created) |

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tokens-management-policy.png" />
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tokens-management-policy.png" />
</div>

## Reviewing Token Authorization

When token policy is set to "Require administrator approval", organization administrators can review details of all fine-grained tokens accessing organization-owned resources and revoke access if needed. Administrators receive email notifications for token authorization requests.

When a token is revoked or denied, the user who created the token receives an email notification.

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tokens-management-review.png" />
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tokens-management-review.png" />
</div>

# Team & Enterprise plans

> [!TIP]
> <a href="https://huggingface.co/enterprise" target="_blank">Subscribe to a Team or Enterprise plan</a> to get access to advanced features for your organization.

Team & Enterprise organization plans add advanced capabilities to organizations, enabling safe, compliant and managed collaboration for companies and teams on Hugging Face.

<a href="https://huggingface.co/enterprise" class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/enterprise-header.png" />
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-enterprise-header.png" />
</a>

In this section we will document the following Enterprise Hub features:

- [Single Sign-On (SSO)](./enterprise-sso)
- [Advanced Single Sign-On (SSO)](./enterprise-hub-advanced-sso)
- [User Provisioning (SCIM)](./enterprise-hub-scim)
- [Audit Logs](./audit-logs)
- [Storage Regions](./storage-regions)
- [Data Studio for Private datasets](./enterprise-hub-datasets)
- [Resource Groups](./security-resource-groups)
- [Advanced Compute Options](./advanced-compute-options)
- [Advanced Security](./enterprise-hub-advanced-security)
- [Tokens Management](./enterprise-hub-tokens-management)
- [Publisher Analytics](./enterprise-hub-analytics)
- [Gating Group Collections](./enterprise-hub-gating-group-collections)
- [Network Security](./enterprise-hub-network-security)
- [Higher Rate limits](./rate-limits)

Finally, Team & Enterprise plans include vastly more [included public storage](./storage-limits), as well as 1TB of [private storage](./storage-limits) per seat in the subscription, i.e. if your organization has 40 members, then you have 40TB included storage for your private models and datasets.

